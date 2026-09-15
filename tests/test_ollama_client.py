# tests/test_ollama_client.py
"""
Regression tests for core/ollama_client.py.

Run with:  pytest tests/test_ollama_client.py -v

`generate()` is the single call site every LLM-touching module routes
through (directly or via core.llm.HestiaLLM). Its contract — never raise,
return "" on any failure, always time and log the call — is exactly the
kind of thing worth locking down with a test, since a regression here
(e.g. letting an exception propagate) would take down every caller.
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ollama_client import generate, generate_stream


def _mock_response(json_body, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_body
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.HTTPError("500")
    return resp


def test_generate_returns_response_text_on_success():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_mock_response({"response": "hello there"}),
    ) as mock_post:
        result = generate("say hi", model="mistral", host="127.0.0.1", port=11434)

    assert result == "hello there"
    mock_post.assert_called_once()
    url = mock_post.call_args.args[0]
    assert url == "http://127.0.0.1:11434/api/generate"


def test_generate_sends_model_and_prompt_in_body():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_mock_response({"response": "ok"}),
    ) as mock_post:
        generate("what time is it", model="llama3", host="h", port=1)

    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "llama3"
    assert body["prompt"] == "what time is it"
    assert body["stream"] is False
    assert "format" not in body
    assert "options" not in body


def test_generate_includes_fmt_and_options_when_provided():
    schema = {"type": "object"}
    with patch(
        "core.ollama_client.requests.post",
        return_value=_mock_response({"response": "{}"}),
    ) as mock_post:
        generate("classify this", fmt=schema, options={"temperature": 0.1})

    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == schema
    assert body["options"] == {"temperature": 0.1}


def test_generate_returns_empty_string_on_connection_error():
    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        result = generate("hello")

    assert result == ""


def test_generate_returns_empty_string_on_timeout():
    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.Timeout("too slow"),
    ):
        result = generate("hello")

    assert result == ""


def test_generate_returns_empty_string_on_http_error_status():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_mock_response({}, status_ok=False),
    ):
        result = generate("hello")

    assert result == ""


def test_generate_returns_empty_string_on_malformed_json_body():
    # Response is JSON but missing the "response" key entirely.
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"unexpected": "shape"}
    with patch("core.ollama_client.requests.post", return_value=resp):
        result = generate("hello")

    assert result == ""


def test_generate_passes_timeout_through_to_requests():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_mock_response({"response": "ok"}),
    ) as mock_post:
        generate("hello", timeout=30)

    assert mock_post.call_args.kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------

class _FakeStreamingResponse:
    """Stand-in for the `with requests.post(..., stream=True) as response`
    context manager, yielding pre-scripted raw lines from iter_lines()."""

    def __init__(self, lines, status_ok=True):
        self._lines = list(lines)
        self._status_ok = status_ok

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.HTTPError("500")

    def iter_lines(self):
        return iter(self._lines)


def _ndjson(*pieces):
    """Build the sequence of raw NDJSON lines Ollama's streaming endpoint
    sends: one {"response": piece, "done": false} per piece, followed by
    a final {"response": "", "done": true}."""
    lines = [
        json.dumps({"response": piece, "done": False}).encode("utf-8")
        for piece in pieces
    ]
    lines.append(json.dumps({"response": "", "done": True}).encode("utf-8"))
    return lines


def test_generate_stream_yields_pieces_in_order():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_FakeStreamingResponse(_ndjson("Hel", "lo ", "world")),
    ) as mock_post:
        result = list(generate_stream("say hi", model="mistral", host="127.0.0.1", port=11434))

    assert result == ["Hel", "lo ", "world"]
    url = mock_post.call_args.args[0]
    assert url == "http://127.0.0.1:11434/api/generate"
    assert mock_post.call_args.kwargs["json"]["stream"] is True


def test_generate_stream_stops_at_done_true_even_with_trailing_lines():
    lines = _ndjson("first") + [
        json.dumps({"response": "should not appear", "done": False}).encode("utf-8")
    ]
    with patch(
        "core.ollama_client.requests.post",
        return_value=_FakeStreamingResponse(lines),
    ):
        result = list(generate_stream("hello"))

    assert result == ["first"]


def test_generate_stream_skips_blank_lines():
    lines = [b""] + _ndjson("a", "b")
    with patch(
        "core.ollama_client.requests.post",
        return_value=_FakeStreamingResponse(lines),
    ):
        result = list(generate_stream("hello"))

    assert result == ["a", "b"]


def test_generate_stream_skips_malformed_json_lines():
    lines = [b"{not json"] + _ndjson("ok")
    with patch(
        "core.ollama_client.requests.post",
        return_value=_FakeStreamingResponse(lines),
    ):
        result = list(generate_stream("hello"))

    assert result == ["ok"]


def test_generate_stream_yields_nothing_on_connection_error():
    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        result = list(generate_stream("hello"))

    assert result == []


def test_generate_stream_yields_partial_output_before_a_mid_stream_failure():
    class _BreaksAfterFirstLine(_FakeStreamingResponse):
        def iter_lines(self):
            yield self._lines[0]
            raise requests.ConnectionError("dropped")

    with patch(
        "core.ollama_client.requests.post",
        return_value=_BreaksAfterFirstLine(_ndjson("partial", "never seen")),
    ):
        result = list(generate_stream("hello"))

    assert result == ["partial"]


def test_generate_stream_yields_nothing_on_http_error_status():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_FakeStreamingResponse(_ndjson("ignored"), status_ok=False),
    ):
        result = list(generate_stream("hello"))

    assert result == []


def test_generate_stream_includes_options_when_provided():
    with patch(
        "core.ollama_client.requests.post",
        return_value=_FakeStreamingResponse(_ndjson("ok")),
    ) as mock_post:
        list(generate_stream("hello", options={"temperature": 0.1}))

    assert mock_post.call_args.kwargs["json"]["options"] == {"temperature": 0.1}