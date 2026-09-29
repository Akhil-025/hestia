# tests/test_chronos_wiring.py
"""
Wiring tests for the Chronos backlog (#81-#90): the heartbeat hand-off, the
Mnemosyne location listener, the intent registry / prompt / alias entries
and config validation.

Run with:  pytest tests/test_chronos_wiring.py -v
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.heartbeat import HestiaHeartbeat  # noqa: E402
from core.intent_aliases import IntentAliasResolver  # noqa: E402
from modules.hecate.intent_registry import ALL_INTENTS, module_for_intent  # noqa: E402

NEW_INTENTS = (
    "list_reminders", "cancel_reminder", "snooze_reminder", "get_agenda",
    "mark_holiday", "unmark_holiday", "save_place", "export_calendar",
    "import_calendar", "weather_plan",
)


# -- heartbeat hand-off ------------------------------------------------------

def test_heartbeat_fires_reminders_by_default():
    mnemosyne = MagicMock()
    mnemosyne.get_due_reminders.return_value = [(1, "take the trash out")]
    hb = HestiaHeartbeat(interval=1800, mnemosyne=mnemosyne)
    assert hb.handle_reminders is True
    hb._run_heartbeat()
    mnemosyne.get_due_reminders.assert_called_once()
    mnemosyne.mark_reminder_done.assert_called_once_with(1)


def test_heartbeat_leaves_reminders_to_chronos_when_disabled():
    mnemosyne = MagicMock()
    mnemosyne.get_due_reminders.return_value = [(1, "take the trash out")]
    hb = HestiaHeartbeat(interval=1800, mnemosyne=mnemosyne)
    hb.handle_reminders = False
    hb._run_heartbeat()
    mnemosyne.get_due_reminders.assert_not_called()
    mnemosyne.mark_reminder_done.assert_not_called()


# -- registry ----------------------------------------------------------------

@pytest.mark.parametrize("intent", NEW_INTENTS)
def test_new_intents_route_to_chronos(intent):
    assert intent in ALL_INTENTS
    assert module_for_intent(intent) == "chronos"


# -- aliases -----------------------------------------------------------------

@pytest.mark.parametrize(
    "query, intent",
    [
        ("what reminders do I have", "list_reminders"),
        ("cancel my reminder about the gym", "cancel_reminder"),
        ("snooze that for 10 minutes", "snooze_reminder"),
        ("snooze", "snooze_reminder"),
        ("what's on my plate today", "get_agenda"),
        ("mark tomorrow as a holiday", "mark_holiday"),
        ("unmark tomorrow as a holiday", "unmark_holiday"),
        ("save my location as home", "save_place"),
        ("export my reminders to a calendar file", "export_calendar"),
        ("import reminders from my calendar file", "import_calendar"),
    ],
)
def test_aliases_resolve_new_intents(query, intent):
    resolver = IntentAliasResolver("config/intent_aliases.yaml")
    assert resolver.dropped == []
    assert resolver.resolve(query) == intent


def test_remind_me_still_goes_to_set_reminder():
    resolver = IntentAliasResolver("config/intent_aliases.yaml")
    assert resolver.resolve("remind me to take my vitamins every weekday at 7am") == "set_reminder"


# -- config validation -------------------------------------------------------

def test_valid_chronos_config_passes():
    from core.config_validation import validate_config
    cfg = {"database": {"path": "x.db"}, "chronos": {
        "timezone": "Asia/Kolkata", "scheduler_enabled": True,
        "scheduler_interval_seconds": 30, "default_snooze_minutes": 10,
        "skip_public_holidays": False, "holiday_country": "IN",
        "proactive_weather": False, "exports_dir": "data/exports"}}
    report = validate_config(cfg)
    assert not [e for e in report.errors if "chronos" in e]


def test_wrongly_typed_chronos_option_is_reported():
    from core.config_validation import validate_config
    report = validate_config({"database": {"path": "x.db"},
                              "chronos": {"scheduler_enabled": "yes"}})
    assert any("chronos.scheduler_enabled" in e for e in report.errors)
