# tests/test_config_validation.py
"""
Tests for core/config_validation.py (backlog #14).

The point of the validator is that a bad config fails at startup naming
the key responsible, instead of failing deep inside a module later with a
message that never mentions config. So these tests assert on the *content*
of the messages, not just on ok/not-ok — a validator that rejects a config
without saying which key is wrong hasn't solved the problem it exists for.

The real config/laptop_config.yaml is also validated, so this file fails
if the shipped config and the schema ever disagree.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config_validation import (
    ConfigError,
    validate_config,
    validate_or_raise,
)

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def minimal_config() -> dict:
    """Smallest config that must validate cleanly."""
    return {
        "ollama": {"model": "mistral", "host": "127.0.0.1", "port": 11434},
        "database": {"path": "data/hestia.db"},
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_minimal_config_is_valid():
    report = validate_config(minimal_config())
    assert report.ok, report.format()
    assert report.errors == []


def test_shipped_config_is_valid():
    # Guards against the schema drifting away from the config that's
    # actually in the repo — the failure mode where adding a required key
    # breaks every existing install.
    path = os.path.join(_REPO_ROOT, "config", "laptop_config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    report = validate_config(cfg)
    assert report.ok, report.format()


def test_shipped_example_config_is_valid():
    path = os.path.join(_REPO_ROOT, "config", "laptop_config.example.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    report = validate_config(cfg)
    assert report.ok, report.format()


def test_optional_sections_absent_is_fine():
    cfg = minimal_config()
    report = validate_config(cfg)
    assert report.ok
    assert not any("webui" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------

def test_missing_ollama_section_is_an_error_naming_the_key():
    cfg = minimal_config()
    del cfg["ollama"]
    report = validate_config(cfg)
    assert not report.ok
    assert any(e.startswith("ollama:") for e in report.errors)


def test_missing_nested_required_key_names_the_full_path():
    cfg = minimal_config()
    del cfg["database"]["path"]
    report = validate_config(cfg)
    assert not report.ok
    assert any("database.path" in e for e in report.errors)


def test_explicit_null_on_a_required_key_is_an_error():
    # `database:\n  path:` is a natural YAML mistake and used to surface as
    # an sqlite3 error about an empty filename.
    cfg = minimal_config()
    cfg["database"]["path"] = None
    report = validate_config(cfg)
    assert not report.ok
    assert any("database.path" in e and "null" in e for e in report.errors)


def test_explicit_null_on_an_optional_key_is_not_an_error():
    cfg = minimal_config()
    cfg["nlu"] = {"model": None}
    assert validate_config(cfg).ok


def test_empty_string_on_a_required_key_is_an_error():
    cfg = minimal_config()
    cfg["database"]["path"] = "   "
    report = validate_config(cfg)
    assert not report.ok
    assert any("empty string" in e for e in report.errors)


def test_all_errors_are_reported_at_once():
    # Reporting one error per run means one restart per mistake.
    cfg = {"ollama": {}, "database": {}}
    report = validate_config(cfg)
    assert len(report.errors) >= 2


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

def test_quoted_port_is_rejected_with_an_actionable_hint():
    cfg = minimal_config()
    cfg["ollama"]["port"] = "11434"
    report = validate_config(cfg)
    assert not report.ok
    message = next(e for e in report.errors if "ollama.port" in e)
    assert "without quotes" in message


def test_string_false_is_rejected_for_a_boolean_flag():
    # The specific trap this validator exists for: the string "false" is
    # truthy in Python, so `webui.enabled: "false"` silently started the
    # web UI anyway.
    cfg = minimal_config()
    cfg["webui"] = {"enabled": "false"}
    report = validate_config(cfg)
    assert not report.ok
    message = next(e for e in report.errors if "webui.enabled" in e)
    assert "truthy" in message


def test_integer_is_rejected_for_a_boolean_flag():
    cfg = minimal_config()
    cfg["webui"] = {"enabled": 1}
    report = validate_config(cfg)
    assert not report.ok
    assert any("webui.enabled" in e for e in report.errors)


def test_boolean_is_rejected_for_an_integer_key():
    # bool is a subclass of int in Python, so isinstance(True, int) is
    # True — without an explicit check this sailed through.
    cfg = minimal_config()
    cfg["ollama"]["port"] = True
    report = validate_config(cfg)
    assert not report.ok
    assert any("ollama.port" in e and "boolean" in e for e in report.errors)


def test_valid_booleans_pass():
    cfg = minimal_config()
    cfg["webui"] = {"enabled": True}
    cfg["browser"] = {"enabled": False, "headless": True}
    assert validate_config(cfg).ok


def test_scalar_where_a_mapping_is_expected():
    cfg = minimal_config()
    cfg["ollama"] = "mistral"
    report = validate_config(cfg)
    assert not report.ok
    assert any("ollama:" in e and "dict" in e for e in report.errors)


def test_out_of_range_port_is_rejected():
    cfg = minimal_config()
    cfg["ollama"]["port"] = 99999
    report = validate_config(cfg)
    assert not report.ok
    assert any("65535" in e for e in report.errors)


def test_port_zero_is_rejected():
    cfg = minimal_config()
    cfg["webui"] = {"port": 0}
    assert not validate_config(cfg).ok


# ---------------------------------------------------------------------------
# Top-level shape and warnings
# ---------------------------------------------------------------------------

def test_non_mapping_config_is_an_error_not_a_crash():
    for value in ([], "text", 42, None):
        report = validate_config(value)
        assert not report.ok
        assert "mapping" in report.errors[0]


def test_unknown_top_level_section_is_a_warning_not_an_error():
    cfg = minimal_config()
    cfg["databse"] = {"path": "typo.db"}      # deliberate typo
    report = validate_config(cfg)
    assert report.ok                           # not fatal
    assert any("databse" in w for w in report.warnings)


def test_known_sections_produce_no_warnings():
    cfg = minimal_config()
    cfg.update({"stt": {}, "tts": {}, "heartbeat": {}, "skills": {}})
    assert validate_config(cfg).warnings == []


# ---------------------------------------------------------------------------
# Reporting and raising
# ---------------------------------------------------------------------------

def test_format_lists_errors_and_warnings():
    cfg = {"ollama": {}, "database": {"path": "x"}, "nonsense": 1}
    text = validate_config(cfg).format()
    assert "ERROR" in text and "WARN" in text


def test_format_of_a_clean_config():
    assert validate_config(minimal_config()).format() == "Configuration OK."


def test_validate_or_raise_passes_through_a_valid_config():
    report = validate_or_raise(minimal_config(), source="test")
    assert report.ok


def test_validate_or_raise_raises_with_the_source_and_details():
    with pytest.raises(ConfigError) as exc:
        validate_or_raise({"database": {"path": "x"}}, source="my_config.yaml")
    message = str(exc.value)
    assert "my_config.yaml" in message
    assert "ollama" in message
