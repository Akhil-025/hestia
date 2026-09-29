# tests/test_heartbeat.py
"""
Regression tests for core/heartbeat.py.

Run with:  pytest tests/test_heartbeat.py -v

HEARTBEAT.md documents that every `- [ ] ` line is a *recurring* condition,
re-evaluated on every tick, whose own in-memory state (`_last_brief_date`,
`_reminder_last_fired`) controls how often it actually fires — the file is
never rewritten. These tests exercise exactly that in-memory state
directly via `_evaluate_task()`, which is what makes the time-window and
cooldown logic testable without a real 30-minute tick loop.

`bus.emit` and wall-clock time are mocked throughout so tests are fast,
deterministic, and don't depend on what time it actually is when the
suite runs.
"""
import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.heartbeat import HestiaHeartbeat


class _FixedNow:
    """Stand-in for the `datetime` module-level name imported into
    core.heartbeat, exposing just the `.now()` call site that module uses.
    """

    def __init__(self, fixed_datetime):
        self._fixed = fixed_datetime

    def now(self):
        return self._fixed


class _FixedToday:
    """Stand-in for the `date` name, exposing just `.today()`."""

    def __init__(self, fixed_date):
        self._fixed = fixed_date

    def today(self):
        return self._fixed


def make_heartbeat(mnemosyne=None, diagnostics=None) -> HestiaHeartbeat:
    return HestiaHeartbeat(interval=1800, mnemosyne=mnemosyne, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Nightly summary — fires only in the 00:00-05:59 window, and only if the
# mnemosyne has a summariser configured.
# ---------------------------------------------------------------------------

def test_nightly_summary_fires_within_window_with_summariser():
    mnemosyne = MagicMock()
    mnemosyne.summariser = object()
    hb = make_heartbeat(mnemosyne)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))):
        hb._evaluate_task("nightly summary")

    mock_bus.emit.assert_called_once_with("mnemosyne_summarise", {})


def test_nightly_summary_does_not_fire_outside_window():
    mnemosyne = MagicMock()
    mnemosyne.summariser = object()
    hb = make_heartbeat(mnemosyne)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(14))):
        hb._evaluate_task("nightly summary")

    mock_bus.emit.assert_not_called()


def test_nightly_summary_does_not_fire_without_summariser():
    mnemosyne = MagicMock()
    mnemosyne.summariser = None
    hb = make_heartbeat(mnemosyne)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))):
        hb._evaluate_task("nightly summary")

    mock_bus.emit.assert_not_called()


def test_nightly_summary_does_not_fire_without_mnemosyne():
    hb = make_heartbeat(mnemosyne=None)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))):
        hb._evaluate_task("nightly summary")

    mock_bus.emit.assert_not_called()


# ---------------------------------------------------------------------------
# Morning brief — fires once per calendar day, only in the 07:00-09:59 window.
# ---------------------------------------------------------------------------

def test_morning_brief_fires_within_window():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(8))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._evaluate_task("morning brief")

    assert mock_bus.emit.call_count >= 1
    emitted_events = [call.args[0] for call in mock_bus.emit.call_args_list]
    assert "morning_brief_requested" in emitted_events


def test_morning_brief_does_not_fire_outside_window():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(15))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._evaluate_task("morning brief")

    mock_bus.emit.assert_not_called()


def test_morning_brief_only_fires_once_per_day():
    hb = make_heartbeat()
    today = date(2026, 1, 1)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(8))), \
         patch("core.heartbeat.date", _FixedToday(today)):
        hb._evaluate_task("morning brief")   # fires
        hb._evaluate_task("morning brief")   # same day: should not re-fire

    # Each firing emits >= 2 events ("speak" x2 + morning_brief_requested);
    # a second identical batch would show up as a doubled call count.
    first_day_calls = mock_bus.emit.call_count
    assert first_day_calls > 0

    with patch("core.heartbeat.bus") as mock_bus2, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(8))), \
         patch("core.heartbeat.date", _FixedToday(today)):
        hb._evaluate_task("morning brief")   # still same day: still no-op

    mock_bus2.emit.assert_not_called()


def test_morning_brief_fires_again_on_a_new_day():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(8))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._evaluate_task("morning brief")

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(8))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 2))):
        hb._evaluate_task("morning brief")

    assert mock_bus.emit.call_count >= 1


# ---------------------------------------------------------------------------
# Reminders — fire immediately, then respect a 4-hour cooldown per task text.
# ---------------------------------------------------------------------------

def test_reminder_fires_on_first_evaluation():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.time.time", return_value=1_000_000.0):
        hb._evaluate_task("reminder: drink some water")

    mock_bus.emit.assert_called_once_with(
        "speak", {"text": "drink some water"}
    )


def test_reminder_does_not_refire_within_cooldown():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.time.time", return_value=1_000_000.0):
        hb._evaluate_task("reminder: drink some water")   # fires
        hb._evaluate_task("reminder: drink some water")   # 0s later: cooldown

    mock_bus.emit.assert_called_once()


def test_reminder_refires_after_cooldown_elapses():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.time.time", return_value=1_000_000.0):
        hb._evaluate_task("reminder: drink some water")

    four_hours = 4 * 3600
    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.time.time", return_value=1_000_000.0 + four_hours + 1):
        hb._evaluate_task("reminder: drink some water")

    mock_bus.emit.assert_called_once_with(
        "speak", {"text": "drink some water"}
    )


def test_reminder_with_empty_text_does_not_fire():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus:
        hb._evaluate_task("reminder:")

    mock_bus.emit.assert_not_called()


def test_reminder_cooldown_is_tracked_per_distinct_task_text():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.time.time", return_value=1_000_000.0):
        hb._evaluate_task("reminder: drink some water")
        hb._evaluate_task("reminder: stretch your legs")

    assert mock_bus.emit.call_count == 2


# ---------------------------------------------------------------------------
# Unrecognised tasks
# ---------------------------------------------------------------------------

def test_unrecognised_task_emits_heartbeat_unhandled_task():
    hb = make_heartbeat()

    with patch("core.heartbeat.bus") as mock_bus:
        hb._evaluate_task("water the office plants")

    mock_bus.emit.assert_called_once_with(
        "heartbeat_unhandled_task", {"task": "water the office plants"}
    )


# ---------------------------------------------------------------------------
# _run_heartbeat — due-reminder handling from Mnemosyne
# ---------------------------------------------------------------------------

def test_run_heartbeat_speaks_and_acks_due_reminders():
    mnemosyne = MagicMock()
    mnemosyne.get_due_reminders.return_value = [(1, "take the trash out")]
    hb = make_heartbeat(mnemosyne)

    # Skip HEARTBEAT.md's own task-line evaluation for this test so only
    # the due-reminders path (which runs first, unconditionally) is under
    # test — otherwise whatever the real HEARTBEAT.md file contains would
    # also fire based on actual wall-clock time, making this test flaky.
    with patch("core.heartbeat.bus") as mock_bus, \
         patch("os.path.exists", return_value=False):
        hb._run_heartbeat()

    mock_bus.emit.assert_any_call(
        "speak", {"text": "Reminder: take the trash out"}
    )
    mnemosyne.mark_reminder_done.assert_called_once_with(1)


def test_run_heartbeat_without_mnemosyne_does_not_raise():
    hb = make_heartbeat(mnemosyne=None)
    with patch("os.path.exists", return_value=False):
        hb._run_heartbeat()  # must not raise


# ---------------------------------------------------------------------------
# Nightly low-confidence review (backlog #6) — once per day, 00:00-05:59,
# only when a Diagnostics is injected. Mirrors the morning-brief tests'
# date-tracking pattern exactly (_last_review_date instead of
# _last_brief_date).
# ---------------------------------------------------------------------------

def test_low_confidence_review_fires_within_window():
    diag = MagicMock()
    diag.write_review_queue.return_value = 3
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()

    diag.write_review_queue.assert_called_once()
    mock_bus.emit.assert_any_call(
        "low_confidence_review_ready",
        {"count": 3, "summary": diag.review_queue_summary.return_value},
    )


def test_low_confidence_review_does_not_fire_outside_window():
    diag = MagicMock()
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(14))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()

    diag.write_review_queue.assert_not_called()


def test_low_confidence_review_does_not_fire_without_diagnostics():
    hb = make_heartbeat(diagnostics=None)
    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()  # must not raise
    mock_bus.emit.assert_not_called()


def test_low_confidence_review_only_fires_once_per_day():
    diag = MagicMock()
    diag.write_review_queue.return_value = 1
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()
        hb._maybe_run_low_confidence_review()

    assert diag.write_review_queue.call_count == 1


def test_low_confidence_review_fires_again_on_a_new_day():
    diag = MagicMock()
    diag.write_review_queue.return_value = 1
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 2))):
        hb._maybe_run_low_confidence_review()

    assert diag.write_review_queue.call_count == 2


def test_low_confidence_review_with_nothing_new_does_not_emit():
    diag = MagicMock()
    diag.write_review_queue.return_value = 0
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()

    mock_bus.emit.assert_not_called()


def test_low_confidence_review_swallows_diagnostics_errors():
    diag = MagicMock()
    diag.write_review_queue.side_effect = RuntimeError("disk full")
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_low_confidence_review()  # must not raise

    mock_bus.emit.assert_not_called()


def test_low_confidence_review_is_invoked_from_run_heartbeat():
    diag = MagicMock()
    diag.write_review_queue.return_value = 0
    hb = make_heartbeat(diagnostics=diag)
    with patch("os.path.exists", return_value=False), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._run_heartbeat()
    diag.write_review_queue.assert_called_once()


# ---------------------------------------------------------------------------
# Weekly per-intent accuracy review (backlog #30) — once every 7 days,
# 00:00-05:59, only when a Diagnostics is injected. Uses a 7-day gap
# rather than a calendar-day flag, unlike the daily jobs above.
# ---------------------------------------------------------------------------

def test_weekly_accuracy_review_fires_within_window():
    diag = MagicMock()
    diag.weekly_accuracy_summary.return_value = "worst: pluto_log_expense ~50%"
    diag.worst_performing_intents.return_value = [
        ("pluto_log_expense", {"total": 4, "flagged_wrong": 2, "accuracy_estimate": 0.5}),
    ]
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()

    diag.weekly_accuracy_summary.assert_called_once()
    assert mock_bus.emit.call_args[0][0] == "weekly_accuracy_review_ready"


def test_weekly_accuracy_review_does_not_fire_outside_window():
    diag = MagicMock()
    hb = make_heartbeat(diagnostics=diag)
    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(14))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()
    diag.weekly_accuracy_summary.assert_not_called()


def test_weekly_accuracy_review_does_not_fire_without_diagnostics():
    hb = make_heartbeat(diagnostics=None)
    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()  # must not raise
    mock_bus.emit.assert_not_called()


def test_weekly_accuracy_review_does_not_repeat_within_seven_days():
    diag = MagicMock()
    diag.worst_performing_intents.return_value = []
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()

    # Day 3: still within the 7-day gap.
    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 4))):
        hb._maybe_run_weekly_accuracy_review()

    assert diag.weekly_accuracy_summary.call_count == 1


def test_weekly_accuracy_review_fires_again_after_seven_days():
    diag = MagicMock()
    diag.worst_performing_intents.return_value = []
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()

    with patch("core.heartbeat.bus"), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 8))):
        hb._maybe_run_weekly_accuracy_review()

    assert diag.weekly_accuracy_summary.call_count == 2


def test_weekly_accuracy_review_with_nothing_flagged_does_not_emit():
    diag = MagicMock()
    diag.weekly_accuracy_summary.return_value = "No mistakes reported."
    diag.worst_performing_intents.return_value = [
        ("chat", {"total": 10, "flagged_wrong": 0, "accuracy_estimate": 1.0}),
    ]
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()

    mock_bus.emit.assert_not_called()


def test_weekly_accuracy_review_swallows_diagnostics_errors():
    diag = MagicMock()
    diag.weekly_accuracy_summary.side_effect = RuntimeError("disk full")
    hb = make_heartbeat(diagnostics=diag)

    with patch("core.heartbeat.bus") as mock_bus, \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._maybe_run_weekly_accuracy_review()  # must not raise

    mock_bus.emit.assert_not_called()


def test_weekly_accuracy_review_is_invoked_from_run_heartbeat():
    diag = MagicMock()
    diag.write_review_queue.return_value = 0
    diag.worst_performing_intents.return_value = []
    hb = make_heartbeat(diagnostics=diag)
    with patch("os.path.exists", return_value=False), \
         patch("core.heartbeat.datetime", _FixedNow(_dt_at_hour(2))), \
         patch("core.heartbeat.date", _FixedToday(date(2026, 1, 1))):
        hb._run_heartbeat()
    diag.weekly_accuracy_summary.assert_called_once()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_at_hour(hour: int):
    from datetime import datetime as _dt
    return _dt(2026, 1, 1, hour, 0, 0)
