# tests/test_main_cli.py
"""
Tests for main.py's CLI surface (backlog #1, #14, #18, #275).

Everything here exercises the pure/near-pure helpers — argument parsing,
verbosity mapping, --check-config, dry-run rendering, and the routing-log
call — without constructing a Hestia instance. Booting Hestia would start
Ollama, a heartbeat thread, a web UI and a mic listener, none of which
belong in a unit test; the parts worth testing are separable from that by
design (that's what `_parse_args(argv)` taking an explicit argv is for).

The Hestia methods that do need an instance (`_log_routing`,
`resolve_only`, `install_signal_handlers`) are tested by calling them
unbound against a lightweight stand-in, which is possible because they
only touch the attributes they declare.
"""
import argparse
import logging
import os
import signal
import sys
import threading
import types

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as main_module


# ---------------------------------------------------------------------------
# Argument parsing (#1, #14, #275)
# ---------------------------------------------------------------------------

def test_defaults_with_no_arguments():
    args = main_module._parse_args([])
    assert args.voice is False
    assert args.dry_run is None
    assert args.check_config is False
    assert args.verbose is False
    assert args.quiet is False
    assert args.config.endswith("laptop_config.yaml")


def test_config_flag_is_respected():
    assert main_module._parse_args(["--config", "/tmp/x.yaml"]).config == "/tmp/x.yaml"


def test_voice_flag():
    assert main_module._parse_args(["--voice"]).voice is True


def test_dry_run_accepts_a_quoted_query():
    args = main_module._parse_args(["--dry-run", "log my sleep"])
    assert args.dry_run == ["log my sleep"]


def test_dry_run_accepts_an_unquoted_multi_word_query():
    # nargs="+" so forgetting the quotes still works.
    args = main_module._parse_args(["--dry-run", "log", "my", "sleep"])
    assert " ".join(args.dry_run) == "log my sleep"


def test_dry_run_requires_a_query():
    with pytest.raises(SystemExit):
        main_module._parse_args(["--dry-run"])


def test_check_config_flag():
    assert main_module._parse_args(["--check-config"]).check_config is True


def test_verbose_and_quiet_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main_module._parse_args(["--verbose", "--quiet"])


@pytest.mark.parametrize("flag", ["--verbose", "-v"])
def test_verbose_flag_forms(flag):
    assert main_module._parse_args([flag]).verbose is True


@pytest.mark.parametrize("flag", ["--quiet", "-q"])
def test_quiet_flag_forms(flag):
    assert main_module._parse_args([flag]).quiet is True


def test_unknown_flag_is_rejected():
    with pytest.raises(SystemExit):
        main_module._parse_args(["--definitely-not-a-flag"])


# ---------------------------------------------------------------------------
# Verbosity mapping (#275)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "argv,expected",
    [([], "normal"), (["--verbose"], "verbose"), (["--quiet"], "quiet")],
)
def test_verbosity_from_args(argv, expected):
    assert main_module._verbosity_from_args(main_module._parse_args(argv)) == expected


def test_verbosity_from_args_tolerates_a_bare_namespace():
    # web_ui.py and the tests build Namespaces without these attributes.
    assert main_module._verbosity_from_args(argparse.Namespace()) == "normal"


@pytest.mark.parametrize(
    "verbosity,expected_level",
    [
        ("quiet", logging.WARNING),
        ("normal", logging.INFO),
        ("verbose", logging.DEBUG),
    ],
)
def test_configure_logging_sets_the_hestia_logger_level(verbosity, expected_level):
    try:
        returned = main_module._configure_logging(verbosity)
        assert returned.level == expected_level
        assert logging.getLogger("hestia").level == expected_level
    finally:
        main_module._configure_logging("normal")


def test_configure_logging_attaches_the_request_id_filter():
    # Without the filter, the format string's %(request_id)s raises a
    # KeyError on any record from a third-party logger.
    try:
        main_module._configure_logging("normal")
        handlers = logging.getLogger().handlers
        assert handlers
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "m", (), None)
        for handler in handlers:
            for f in handler.filters:
                f.filter(record)
        assert hasattr(record, "request_id")
    finally:
        main_module._configure_logging("normal")


def test_configure_logging_is_idempotent():
    # basicConfig(force=True) means repeated calls must not stack handlers.
    main_module._configure_logging("normal")
    before = len(logging.getLogger().handlers)
    main_module._configure_logging("verbose")
    main_module._configure_logging("normal")
    assert len(logging.getLogger().handlers) == before


# ---------------------------------------------------------------------------
# --check-config (#14)
# ---------------------------------------------------------------------------

def test_check_config_returns_zero_for_the_shipped_config():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "laptop_config.yaml")
    assert main_module._run_check_config(path) == 0


def test_check_config_returns_one_for_a_missing_file(tmp_path):
    assert main_module._run_check_config(str(tmp_path / "nope.yaml")) == 1


def test_check_config_returns_one_for_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("ollama: [unclosed\n", encoding="utf-8")
    assert main_module._run_check_config(str(path)) == 1


def test_check_config_returns_one_for_a_config_missing_required_keys(tmp_path, capsys):
    path = tmp_path / "partial.yaml"
    path.write_text(yaml.safe_dump({"database": {"path": "x.db"}}), encoding="utf-8")
    assert main_module._run_check_config(str(path)) == 1
    # The whole point is that the offending key is named.
    assert "ollama" in capsys.readouterr().out


def test_check_config_prints_the_registry_version(tmp_path, capsys):
    path = os.path.join(os.path.dirname(__file__), "..", "config", "laptop_config.yaml")
    main_module._run_check_config(path)
    assert "Intent registry v" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Dry-run rendering (#1)
# ---------------------------------------------------------------------------

def sample_dry_run(**overrides):
    result = {
        "query": "log my sleep",
        "request_id": "abc12345",
        "intent": "apollo_track_sleep",
        "entities": {},
        "confidence": 0.92,
        "nlu_source": "alias",
        "nlu_latency_ms": 3.2,
        "registry_module": "apollo",
        "primary": "apollo",
        "secondary": [],
        "reason": "registry: intent 'apollo_track_sleep' -> apollo",
        "dispatch_intent": "track_sleep",
        "primary_can_handle": True,
        "synthesize": False,
        "active_modules": ["core", "apollo"],
        "executed": False,
    }
    result.update(overrides)
    return result


def test_format_dry_run_includes_the_key_fields():
    text = main_module.Hestia.format_dry_run(sample_dry_run())
    for fragment in ("log my sleep", "apollo_track_sleep", "apollo", "abc12345"):
        assert fragment in text


def test_format_dry_run_states_that_nothing_ran():
    assert "no handler executed" in main_module.Hestia.format_dry_run(sample_dry_run())


def test_format_dry_run_reports_an_error_result():
    text = main_module.Hestia.format_dry_run({"error": "NLU failed: timeout"})
    assert "dry-run failed" in text
    assert "timeout" in text


def test_format_dry_run_flags_a_registry_routing_disagreement():
    # The single most useful thing a dry run can surface.
    text = main_module.Hestia.format_dry_run(
        sample_dry_run(registry_module="apollo", primary="core")
    )
    assert "NOTE" in text
    assert "registry maps this intent to 'apollo'" in text


def test_format_dry_run_does_not_flag_agreement():
    assert "NOTE: registry maps" not in main_module.Hestia.format_dry_run(
        sample_dry_run()
    )


def test_format_dry_run_flags_a_can_handle_rejection():
    text = main_module.Hestia.format_dry_run(
        sample_dry_run(primary_can_handle=False)
    )
    assert "can_handle() rejects" in text


def test_format_dry_run_does_not_flag_unknown_can_handle():
    # None means "couldn't probe", which is not the same as a rejection.
    text = main_module.Hestia.format_dry_run(sample_dry_run(primary_can_handle=None))
    assert "can_handle() rejects" not in text


def test_format_dry_run_handles_an_unregistered_intent():
    text = main_module.Hestia.format_dry_run(
        sample_dry_run(registry_module=None, primary="core")
    )
    assert "None" in text
    assert "NOTE: registry maps" not in text   # nothing to disagree with


# ---------------------------------------------------------------------------
# _log_routing (#5)
# ---------------------------------------------------------------------------

class _RecordingDiagnostics:
    def __init__(self):
        self.calls = []

    def record_classification(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def _stub_hestia(decision):
    """Minimal stand-in exposing only what _log_routing touches."""
    return types.SimpleNamespace(
        diagnostics=_RecordingDiagnostics(),
        orchestrator=types.SimpleNamespace(last_decision=decision),
    )


def test_log_routing_uses_the_orchestrators_actual_decision():
    # Not a second Hecate call: the log must record where the query really
    # went, including via fallback tiers.
    stub = _stub_hestia({"primary": "apollo", "reason": "registry"})
    main_module.Hestia._log_routing(
        stub,
        "log my sleep",
        {"intent": "apollo_track_sleep", "confidence": 0.92, "source": "alias"},
        41.5,
    )
    call = stub.diagnostics.calls[0]
    assert call["module"] == "apollo"
    assert call["reason"] == "registry"
    assert call["intent"] == "apollo_track_sleep"
    assert call["confidence"] == 0.92
    assert call["latency_ms"] == 41.5
    assert call["source"] == "alias"


def test_log_routing_defaults_when_no_decision_is_recorded():
    stub = _stub_hestia(None)
    main_module.Hestia._log_routing(stub, "hello", {}, 1.0)
    call = stub.diagnostics.calls[0]
    assert call["module"] == "unknown"
    assert call["intent"] == "chat"
    assert call["source"] == "nlu"


# ---------------------------------------------------------------------------
# Signal handlers (#18)
# ---------------------------------------------------------------------------

def test_install_signal_handlers_registers_sigterm_and_sigint():
    shutdowns = []
    stub = types.SimpleNamespace(_shutdown=lambda: shutdowns.append(True))
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        main_module.Hestia.install_signal_handlers(stub)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handler = signal.getsignal(sig)
            assert callable(handler)
            # Invoke it directly rather than actually signalling the test
            # runner; SystemExit is the documented contract.
            with pytest.raises(SystemExit):
                handler(sig, None)
        assert len(shutdowns) == 2
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def test_install_signal_handlers_is_a_noop_off_the_main_thread():
    # Python only allows signal registration on the main thread; the web
    # UI and test suite import main from worker threads.
    stub = types.SimpleNamespace(_shutdown=lambda: None)
    errors = []

    def run():
        try:
            main_module.Hestia.install_signal_handlers(stub)
        except Exception as exc:       # pragma: no cover - the bug case
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert errors == []
