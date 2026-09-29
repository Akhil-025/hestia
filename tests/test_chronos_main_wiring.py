# tests/test_chronos_main_wiring.py
"""
Tests that main.py hands Chronos its configuration and sibling modules, starts
and stops its scheduler, and stops the heartbeat firing reminders twice
(backlog #81-#89).

Every engine except ChronosEngine is replaced with a BaseModule-shaped mock, so
this needs no Ollama, hardware or optional dependencies.

Run with:  pytest tests/test_chronos_main_wiring.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main  # noqa: E402
from modules.base import BaseModule  # noqa: E402

_OTHER_ENGINES = (
    "ArtemisEngine", "HermesEngine", "HephaestusEngine", "ApolloEngine", "AresEngine",
    "OrpheusEngine", "MetisEngine", "DionysusEngine", "PlutoEngine",
)


def _module_mock(name):
    m = MagicMock(spec=BaseModule)
    m.name = name
    m.can_handle.return_value = False
    return m


def _build(chronos_cfg, google_agent=None):
    made = {}

    def factory(cls_name):
        def _f(*a, **k):
            made[cls_name] = _module_mock(cls_name.replace("Engine", "").lower())
            return made[cls_name]
        return _f

    patches = [patch.object(main, n, side_effect=factory(n)) for n in _OTHER_ENGINES]
    for p in patches:
        p.start()
    try:
        builder = main.HestiaBuilder({"chronos": chronos_cfg})
        result = builder.build_orchestrator(
            _module_mock("mnemosyne"),
            {"athena": None, "iris": None, "google_agent": google_agent, "browser_agent": None},
        )
    finally:
        for p in patches:
            p.stop()
    return result, made


def test_chronos_receives_its_config():
    (_, _, _, _, chronos), _ = _build({
        "timezone": "Asia/Tokyo", "skip_public_holidays": True, "holiday_country": "JP",
        "default_snooze_minutes": 5, "proactive_weather": True,
    })
    assert getattr(chronos._tz, "key", None) == "Asia/Tokyo"
    assert chronos._proactive_weather is True


def test_chronos_defaults_when_options_are_missing():
    (_, _, _, _, chronos), _ = _build({})
    assert chronos._proactive_weather is False
    assert getattr(chronos._tz, "key", None) == "Asia/Kolkata"


def test_chronos_is_handed_artemis_and_dionysus():
    (_, _, _, artemis, chronos), made = _build({})
    assert chronos._artemis is artemis is made["ArtemisEngine"]
    assert chronos._dionysus is made["DionysusEngine"]


def test_chronos_is_handed_hermes_when_google_is_configured():
    (_, _, _, _, chronos), made = _build({}, google_agent=MagicMock())
    assert chronos._hermes is made["HermesEngine"]


def test_chronos_has_no_hermes_without_google():
    (_, _, _, _, chronos), _ = _build({})
    assert chronos._hermes is None


def test_chronos_is_registered_with_the_orchestrator():
    (orch, _, _, _, chronos), _ = _build({})
    assert "chronos" in orch.registered_modules


# -- shutdown ----------------------------------------------------------------

def test_shutdown_stops_the_chronos_scheduler():
    h = object.__new__(main.Hestia)
    for attr in ("barge_in", "heartbeat", "chronos"):
        setattr(h, attr, MagicMock())
    h._stop_hot_reload_watchers = MagicMock()
    with patch.object(main, "bus", MagicMock()):
        h._shutdown()
    h.chronos.stop_scheduler.assert_called_once()
    h.heartbeat.stop.assert_called_once()


def test_shutdown_survives_a_scheduler_that_raises():
    h = object.__new__(main.Hestia)
    for attr in ("barge_in", "heartbeat", "chronos"):
        setattr(h, attr, MagicMock())
    h._stop_hot_reload_watchers = MagicMock()
    h.chronos.stop_scheduler.side_effect = RuntimeError("boom")
    with patch.object(main, "bus", MagicMock()):
        h._shutdown()          # must not raise
    h.heartbeat.stop.assert_called_once()
