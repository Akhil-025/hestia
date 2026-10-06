# tests/test_main_roles.py
"""main.py wiring for backlog #4 / #16 / #20: CLI flags, builders, role-aware speech."""
import os
import sys
import types

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as main_module
from core.event_queue import EventQueue
from core.process_split import SAY_TOPIC, profile_for


# ---------------------------------------------------------------- CLI

def test_role_defaults_to_all():
    assert main_module._parse_args([]).role == "all"


def test_role_accepts_the_known_values_and_rejects_others():
    for r in ("all", "core", "voice", "jobs", "supervisor"):
        assert main_module._parse_args(["--role", r]).role == r
    with pytest.raises(SystemExit):
        main_module._parse_args(["--role", "bogus"])


def test_new_flags_parse():
    a = main_module._parse_args(["--train-classifier", "--shadow-report", "--label", "log my sleep", "apollo_track_sleep"])
    assert a.train_classifier and a.shadow_report and a.label == ["log my sleep", "apollo_track_sleep"]


def test_label_command_writes_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main_module._run_label("log my sleep", "apollo_track_sleep") == 0
    assert (tmp_path / "data" / "classifier_labels.jsonl").exists()
    assert main_module._run_label("x", "no_such_intent") == 1


def test_shadow_report_with_no_rules(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"ollama": {"model": "m"}, "database": {"path": "x.db"}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main_module._run_shadow_report(str(cfg)) == 0


# ---------------------------------------------------------------- builders

def test_classifier_is_off_unless_configured():
    assert main_module.HestiaBuilder({}).build_classifier() is None
    assert main_module.HestiaBuilder({"classifier": {"mode": "off"}}).build_classifier() is None


def test_bare_yaml_off_means_off():
    cfg = yaml.safe_load("classifier:\n  mode: off\n")
    assert cfg["classifier"]["mode"] is False        # the YAML gotcha
    assert main_module.HestiaBuilder(cfg).build_classifier() is None


def test_unknown_classifier_mode_leaves_it_off():
    assert main_module.HestiaBuilder({"classifier": {"mode": "turbo"}}).build_classifier() is None


def test_shipped_example_config_validates_with_the_new_blocks():
    from core.config_validation import validate_config
    with open(os.path.join(os.path.dirname(__file__), "..", "config", "laptop_config.example.yaml"),
              encoding="utf-8") as fh:
        assert validate_config(yaml.safe_load(fh)).ok


def test_classifier_builds_in_assist_mode(tmp_path):
    cfg = {"classifier": {"mode": "assist", "model_path": str(tmp_path / "m.npz")}}
    svc = main_module.HestiaBuilder(cfg).build_classifier()
    assert svc is not None and svc.mode == "assist"


def test_shadow_off_without_enabled_rules():
    assert main_module.HestiaBuilder({}).build_shadow() is None
    assert main_module.HestiaBuilder({"shadow": {"enabled": True, "rules": []}}).build_shadow() is None


def test_shadow_builds_when_a_valid_rule_is_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"shadow": {"enabled": True, "rules": [{
        "name": "r", "intent": "get_weather", "candidate": {"module": "m", "intent": "i"},
        "mode": "shadow", "read_only": True}]}}
    rec = main_module.HestiaBuilder(cfg).build_shadow()
    assert rec is not None and len(rec.rules.rules) == 1


def test_build_io_only_builds_the_requested_components(monkeypatch):
    built = []

    def fake(name):
        class F:
            def __init__(self, *a, **k): built.append(name)
        return F
    monkeypatch.setattr(main_module, "HestiaSTT", fake("stt"))
    monkeypatch.setattr(main_module, "HestiaTTS", fake("tts"))
    monkeypatch.setattr(main_module, "WakeWordDetector", fake("wake"))
    monkeypatch.setattr(main_module, "BargeInListener", fake("barge"))
    b = main_module.HestiaBuilder({})
    stt, tts, wake, barge = b.build_io(only=frozenset({"stt"}))
    assert built == ["stt"]
    assert stt is not None and wake is None and barge is None
    assert isinstance(tts, main_module.NullTTS)
    assert b.io_errors == {}                      # skipping is not an error


# ---------------------------------------------------------------- _speak by role

class _TTS:
    def __init__(self): self.said = []
    def speak(self, text, voice=None): self.said.append((text, voice))


def _hestia_stub(role, tmp_path):
    stub = types.SimpleNamespace(role=role, tts=_TTS())
    stub.profile = profile_for(role)
    stub._queue = EventQueue(tmp_path / "e.db", origin=role) if role != "all" else None
    return stub


def test_all_role_speaks_locally_exactly_as_before(tmp_path):
    h = _hestia_stub("all", tmp_path)
    main_module.Hestia._speak(h, "hello")
    main_module.Hestia._speak(h, "hi", "calm")
    assert h.tts.said == [("hello", None), ("hi", "calm")]


def test_core_role_queues_speech_for_the_voice_process(tmp_path):
    h = _hestia_stub("core", tmp_path)
    probe = EventQueue(tmp_path / "e.db", origin="probe")
    probe.start_from_beginning("p")
    main_module.Hestia._speak(h, "hello", "calm")
    ev = probe.consume("p", [SAY_TOPIC])[0]
    assert ev["payload"] == {"text": "hello", "voice": "calm"} and h.tts.said == []


def test_jobs_role_hands_speech_to_core_as_a_notification(tmp_path):
    got = []
    main_module.bus.on("speak", got.append)
    try:
        h = _hestia_stub("jobs", tmp_path)
        main_module.Hestia._speak(h, "Reminder: stretch")
        import time
        end = time.time() + 2
        while time.time() < end and not got:
            time.sleep(0.02)
    finally:
        main_module.bus.off("speak", got.append)
    assert got and got[0]["text"] == "Reminder: stretch" and h.tts.said == []


def test_bare_instance_defaults_to_the_single_process():
    assert main_module.Hestia.role == "all" and main_module.Hestia.profile is None
