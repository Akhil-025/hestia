# tests/test_ollama_manager.py
"""
Regression tests for core/ollama_manager.py.

Run with:  pytest tests/test_ollama_manager.py -v

All network calls (requests.get) and process spawning (subprocess.Popen)
are mocked — these tests never touch a real Ollama instance or spawn a
real subprocess, so they run the same whether or not Ollama is installed.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ollama_manager import OllamaManager


def make_manager() -> OllamaManager:
    return OllamaManager(host="127.0.0.1", port=11434)


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------

def test_is_running_true_on_200():
    mgr = make_manager()
    mock_response = MagicMock(status_code=200)
    with patch("core.ollama_manager.requests.get", return_value=mock_response):
        assert mgr.is_running() is True


def test_is_running_false_on_non_200():
    mgr = make_manager()
    mock_response = MagicMock(status_code=500)
    with patch("core.ollama_manager.requests.get", return_value=mock_response):
        assert mgr.is_running() is False


def test_is_running_false_on_connection_error():
    mgr = make_manager()
    with patch(
        "core.ollama_manager.requests.get",
        side_effect=requests.ConnectionError("refused"),
    ):
        assert mgr.is_running() is False


def test_is_running_false_on_timeout():
    mgr = make_manager()
    with patch(
        "core.ollama_manager.requests.get",
        side_effect=requests.Timeout("slow"),
    ):
        assert mgr.is_running() is False


def test_is_running_does_not_swallow_keyboard_interrupt():
    # The bare `except:` this was narrowed from used to also catch
    # KeyboardInterrupt/SystemExit; requests.RequestException must not.
    mgr = make_manager()
    with patch("core.ollama_manager.requests.get", side_effect=KeyboardInterrupt):
        try:
            mgr.is_running()
        except KeyboardInterrupt:
            pass
        else:
            assert False, "KeyboardInterrupt should propagate, not be swallowed"


def test_base_url_is_built_from_host_and_port():
    mgr = OllamaManager(host="192.168.1.5", port=1234)
    assert mgr.base_url == "http://192.168.1.5:1234"


# ---------------------------------------------------------------------------
# ensure_running
# ---------------------------------------------------------------------------

def test_ensure_running_returns_true_immediately_if_already_running():
    mgr = make_manager()
    with patch.object(mgr, "is_running", return_value=True), \
         patch("core.ollama_manager.subprocess.Popen") as mock_popen:
        assert mgr.ensure_running() is True
    mock_popen.assert_not_called()


def test_ensure_running_starts_process_when_not_running():
    mgr = make_manager()
    # First call (initial check) False, then True once "started".
    with patch.object(mgr, "is_running", side_effect=[False, True]), \
         patch("core.ollama_manager.subprocess.Popen") as mock_popen, \
         patch("core.ollama_manager.time.sleep"):
        result = mgr.ensure_running(retries=1, delay=0)

    assert result is True
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert args[0] == ["ollama", "serve"]


def test_ensure_running_returns_false_if_popen_raises():
    mgr = make_manager()
    with patch.object(mgr, "is_running", return_value=False), \
         patch("core.ollama_manager.subprocess.Popen", side_effect=OSError("not found")):
        assert mgr.ensure_running() is False


def test_ensure_running_returns_false_if_never_comes_up():
    mgr = make_manager()
    with patch.object(mgr, "is_running", return_value=False), \
         patch("core.ollama_manager.subprocess.Popen"), \
         patch("core.ollama_manager.time.sleep"):
        result = mgr.ensure_running(retries=2, delay=0)

    assert result is False
